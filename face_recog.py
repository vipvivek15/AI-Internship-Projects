# import the necessary packages
import face_recognition
from collections import Counter
import pickle
import cv2
from PIL import Image
import os
import sys
import numpy as np
import base64

def identify_face():
        Out_fold = r'E:\facerecog own'

        # load the known faces and embeddings
        print("[INFO] loading encodings...")
        data = pickle.loads(open(r"E:/facerecog_own/face_enc.pickle", "rb").read())
        inti = dict(Counter(data["names"]))
        inti['Unknown'] = 1
        print(data["names"])
        
        # load the input image and convert it from BGR to RGB
        ##/home/aiml/ml/share/data/face_recog/examples
        image = cv2.imread(r"E:\\face_recgn\\dataset\\obama\\obama2.jpg")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
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

    
        # show the output image
        result = Image.fromarray(image)
        result.save(os.path.join(Out_fold, 'out.jpg'))
        return names
        print('file saved..')

def read_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    return image

def prepare_image(image):
    # Create string encoding of the image
    #image_content = cv2.imencode('.jpg', image)[1].tostring()
    # Create base64 encoding of the string encoded image
    encoded_image = base64.encodestring(image)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send