import os
from flask import Flask,jsonify,request,render_template
from source.face_detection import detect_faces_with_ssd
from source.utils import draw_rectangles, read_image, prepare_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Detect faces
    faces = detect_faces_with_ssd(image)

    return jsonify(detections = faces)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Detect faces
    faces = detect_faces_with_ssd(image)
    
    # Draw detection rects
    num_faces, image = draw_rectangles(image, faces)
    
    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('index.html', face_detected=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)

if __name__ == '__main__':
    app.run(host='192.168.1.233',port=9898)
