import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import urllib.request
from mtcnn import MTCNN
import keras
from joblib import dump, load
import playsound
from pygame import mixer
import time




# cap = cv2.VideoCapture("http://192.168.1.245:81/stream")
#
# while True:
#     try:
#         success, img = cap.read()
#
#         cv2.imshow('Image', img)
#         # cv2.waitKey(1)
#     except Exception as e:
#         print("Exception happened: {}".format(e))
#         continue
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# cv2.destroyAllWindows()
def predict(model, img):
    img = cv2.resize(img,(56,56))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = enc.inverse_transform(pred)[0]
    return pred


enc = load('encoder.joblib')
model_recognize = keras.saving.load_model('model.keras')
detector = MTCNN()
while True:
    imgresponce = urllib.request.urlopen("http://192.168.1.245/capture?")
    imgNp = np.array(bytearray(imgresponce.read()),dtype = np.uint8)
    frame = cv2.imdecode(imgNp,-1)
    faces = detector.detect_faces(frame)
    name = ''
    if len(faces) >= 1:
        for face in faces:
            x_min,y_min,width,height = face['box']
            x_max = min(x_min + width,frame.shape[1])
            y_max = min(y_min + height,frame.shape[0])
            x_min = max(x_min,0)
            y_min = max(y_min,0)
            coeff = face['confidence']
            img = frame[y_min:y_max, x_min:x_max, :]
            name = predict(model_recognize, img)[0]
            cv2.rectangle(frame,(x_min,y_min), (x_max,y_max),(0,255,0))
            cv2.putText(frame,f'{coeff}',(x_min+100,y_min),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            cv2.putText(frame, f'{name}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    cv2.imshow('Window',frame)
    if name == 'Thu':
        mixer.init()
        mixer.music.load('C:/Users/Buu/PycharmProjects/Camerawebserver/sound.mp3')  # Loading Music File
        mixer.music.play()  # Playing Music with Pygame
        time.sleep(5)
        mixer.music.stop()
        # playsound.playsound('C:/Users/Buu/PycharmProjects/Camerawebserver/sound.mp3', True)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()