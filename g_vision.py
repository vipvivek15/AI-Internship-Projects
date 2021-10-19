from flask import Flask, render_template, request, redirect, make_response, send_file, session, Markup
from flask_bootstrap import Bootstrap

from google.cloud import vision
from google.cloud import storage
import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccount_DocumentAI.json' 

app = Flask(__name__)
app.secret_key = "Sravan_789" 
bootstrap = Bootstrap(app)

@app.route('/',methods=['GET', 'POST'])  
def g_vision():  
    return render_template("g_vision.html") 
@app.route('/g_vision1',methods=['GET', 'POST'])  
def g_vision1():
    
    if request.form.get("gv_1"):
        gv_title="Label Detection"
        gv_description="The Label Detection can detect and extract information about entities in an image, across a broad group of categories. Labels can identify general objects, locations, activities, animal species, products, and more."
        
    elif request.form.get("gv_2"):
        gv_title="Text Detection"
        gv_description="Text Detection detects and extracts text from any image. For example, a photograph might contain a street sign or traffic sign"
        
    elif request.form.get("gv_3"):
        gv_title="Ladmark Detection"
        gv_description="Landmark Detection detects popular natural and human-made structures within an image."
        
    elif request.form.get("gv_4"):
        gv_title="Logo Detection"
        gv_description="Logo Detection detects popular product logos within an image"
        
    elif request.form.get("gv_5"):
        gv_title="Safesearch Detection"
        gv_description="SafeSearch Detection detects explicit content such as adult content or violent content within an image. This feature uses five categories (adult, spoof, medical, violence, and racy) and returns the likelihood that each is present in a given image."
        
    elif request.form.get("gv_6"):
        gv_title="Emotion Detection"
        gv_description="Emotion Detection detects the emotion of the person in the image. This feature uses four categories (joy, anger, sorrow, and surprise.) and returns the likelihood that each is present in a given image. "
        
    elif request.form.get("gv_7"):
        gv_title="Web Content Detection"
        gv_description="Web Content Detection detects Web references to an image."    
    
    session['gv_title']=gv_title
    session['gv_description']=gv_description
    
    return render_template("g_vision1.html",gv_title=gv_title,gv_description=gv_description) 

@app.route('/g_vision2',methods=['GET', 'POST'])  
def g_vision2():  
    
    gv_title=session['gv_title']
    gv_description=session['gv_description']
    
    file = request.files['file'] 
    filename = file.filename    
     
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'ServiceAccount_DocumentAI.json'
    
    client1 = storage.Client()
    bucket = client1.bucket('sravan_vision')
    blob = bucket.blob(filename)
    blob.upload_from_file(file)
    
    image_uri = "gs://sravan_vision/%s" % (filename)
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = image_uri
    
    if gv_title=="Label Detection":
        
        response1 = client.label_detection(image=image)
        l1=[]
        for label in response1.label_annotations:
            l1.append([label.description, '%.2f%%' % (label.score*100.)])
        return render_template("g_vision1.html",l1=l1,gv_title=gv_title,gv_description=gv_description) 

    elif gv_title=="Text Detection":
        
        response2 = client.text_detection(image=image)
        l2=response2.full_text_annotation.text
        return render_template("g_vision1.html",l2=l2,gv_title=gv_title,gv_description=gv_description) 
  
    elif gv_title=="Ladmark Detection":
        
        response3 = client.landmark_detection(image=image)
        l3=[]
        for landmark in response3.landmark_annotations:
            for landmark2 in landmark.locations:
                l3.append([landmark.description,landmark2.lat_lng])
        return render_template("g_vision1.html",l3=l3,gv_title=gv_title,gv_description=gv_description) 
  
    elif gv_title=="Logo Detection":
        
        response4 = client.logo_detection(image=image)
        l4=[]
        for logo in response4.logo_annotations:
            l4.append([logo.description,'%.2f%%' % (logo.score*100.)])
        return render_template("g_vision1.html",l4=l4,gv_title=gv_title,gv_description=gv_description) 
  
    elif gv_title=="Safesearch Detection":
        
        response5 = client.safe_search_detection(image=image)
        safe = response5.safe_search_annotation
           
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                               'LIKELY', 'VERY_LIKELY')
        l5 = {'Adult': format(likelihood_name[safe.adult]),
                 'Medical': format(likelihood_name[safe.medical]),
                 'Spoofed': format(likelihood_name[safe.spoof]),
                 'Violence': format(likelihood_name[safe.violence]),
                 'Racy': format(likelihood_name[safe.racy])
                }
        return render_template("g_vision1.html",l5=l5,gv_title=gv_title,gv_description=gv_description) 
  
    elif gv_title=="Emotion Detection":
        
        response6 = client.face_detection(image=image)
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                               'LIKELY', 'VERY_LIKELY')
        for face in response6.face_annotations:
            l6 = {'Anger': format(likelihood_name[face.anger_likelihood]),
                 'Joy': format(likelihood_name[face.joy_likelihood]),
                 'Surprise': format(likelihood_name[face.surprise_likelihood]),         
                 'Sorrow': format(likelihood_name[face.sorrow_likelihood])
                }
        return render_template("g_vision1.html",l6=l6,gv_title=gv_title,gv_description=gv_description) 
  
    elif gv_title=="Web Content Detection":
       
        response7 = client.web_detection(image=image)
        annotations = response7.web_detection
        l7=[]
        
        if annotations.best_guess_labels:
            for label in annotations.best_guess_labels:
                l7.append('\nBest guess label: {}'.format(label.label))
        
        if annotations.pages_with_matching_images:
            l7.append('\n{} Pages with matching images found:'.format(
                len(annotations.pages_with_matching_images)))
        
            for page in annotations.pages_with_matching_images:
                l7.append(format(page.url))
        
        return render_template("g_vision1.html",l7=l7,gv_title=gv_title,gv_description=gv_description) 
                
    
if __name__ == '__main__':
    #app.run(port=5000,debug=True)
    app.run(host='192.168.1.233',port=9898)