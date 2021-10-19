# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:34:48 2021

@author: Raghavendra
"""


import os
from flask import Flask,jsonify,request,render_template
import face_recognition
from collections import Counter
import pickle
import cv2
from PIL import Image
import os
import sys
import numpy as np
from face_recog import identify_face , read_image, prepare_image
import base64

app = Flask(__name__)

#UPLOAD_FOLDER = os.path.basename('uploads')
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image,(224,224))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #pil_image = Image.open(file)
    #image = cv2.imdecode(np.fromstring(pil_image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #Out_fold = r'C:\Users\himakar\Desktop\facerecog_own'
    data = pickle.loads(open(r"C:\Users\himakar\Desktop\facerecog_own\face_enc1.pickle", "rb").read())
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb,model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

 
    for encoding in encodings:
         matches = face_recognition.compare_faces(data["encodings"],encoding)
         name = "Unknown"
         if True in matches:
            matchedIdxs  = [i for (i, b) in enumerate(matches) if b]
            counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
            for i in matchedIdxs:
                  name = data["names"][i]
                  counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
	
         names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	        y = top - 15 if top - 15 > 15 else top + 15
	        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

    #picra =Image.frombytes(image)
    #result.save(os.path.join(Out_fold, 'out.jpg'))
    print('raghu2')
    result = Image.fromarray(image)
    imaggera = np.array(result)
    #ingrag = imaggera.ravel()
    image_content = cv2.imencode('.jpg', imaggera)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:encoded_image/jpg;base64,' + str(encoded_image, 'utf-8')
    #to_send = prepare_image(result)
       
    return render_template('index.html',image_to_show=to_send, init=True)


if __name__ == '__main__':
    app.run(host="192.168.1.233",port=9898)