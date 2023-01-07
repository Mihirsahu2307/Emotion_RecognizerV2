from flask import Flask, render_template, request, make_response
import cv2
import numpy as np
from keras.utils import img_to_array
from keras.models import load_model

app = Flask(__name__, template_folder='./templates')

global fps, prev_recv_time, face_roi, emotion_detect, fd_model, status, counter

counter = 0
modelFile = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "face_detection_model/deploy.prototxt.txt"
fd_model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
fps=5
prev_recv_time = 0
emotion_detect = 1
face_roi = np.zeros((3, 3, 3))
status = 'neutral'

# from keras.models import model_from_json
# model = model_from_json(open("prediction_model2/facial_expression_model_structure.json", "r").read())
# model.load_weights('prediction_model2/facial_expression_model_weights.h5')

emotion_model_path = 'prediction_model1/_mini_XCEPTION.102-0.66.hdf5'
model = load_model(emotion_model_path, compile=False)

def predict():
    global face_roi, status
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) #transform to gray scale
    roi = cv2.resize(face_roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    preds = model.predict(roi)[0]
    status = classes[preds.argmax()]


# def predict():
#     global face_roi, status
#     classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
#     face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) #transform to gray scale
#     face_roi = cv2.resize(face_roi, (48, 48)) #resize to 48x48
#     img_pixels = img_to_array(face_roi)
#     img_pixels = np.expand_dims(img_pixels, axis = 0)

#     img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
#     predictions = model.predict(img_pixels) #store probabilities of 7 expressions
#     max_index = np.argmax(predictions[0])
#     status = classes[max_index]

def detect_face(frame):
    global fd_model, face_roi
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    fd_model.setInput(blob)
    detections = fd_model.forward()
    confidence = detections[0, 0, 0, 2] # atmost 1 face detected

    if confidence < 0.5:            
        return (frame, -1, -1)           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (x, y, x1, y1) = box.astype("int")
    try:
        # dim = (h, w)
        face_roi = frame[y:y1, x:x1]
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        raise
    
    return (frame, x, y)

def send_file_data(data, mimetype='image/jpeg', filename='output.jpg'):
    response = make_response(data)
    response.headers.set('Content-Type', mimetype)
    response.headers.set('Content-Disposition', 'attachment', filename=filename)

    return response

@app.route('/webcam', methods = ['GET', 'POST'])
def process_image():
    global prev_recv_time, counter, status
    image = request.files.get('image')
    
    try:
        frame = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        frame = cv2.flip(frame,1)
        frame, x, y = detect_face(frame)
        
        if(emotion_detect and x > -1 and y > -1):
            predict()
            cv2.putText(frame, status, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
    except:
        raise
    
    buf = cv2.imencode('.jpg', frame)[1]
    return send_file_data(buf.tobytes())
    

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000 ,debug=True)
