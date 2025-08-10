import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import cv2
import os
import datetime
import re
import urllib.request
import warnings
from yolov5facedetector.face_detector import YoloDetector

warnings.filterwarnings("ignore")
detector = YoloDetector(gpu=0, min_face = 28)
while True:

    # responce = urllib.request.urlopen("http://192.168.1.245/capture?")
    # imgNP = np.array(bytearray(responce.read()), dtype= np.uint8)
    # frame = cv2.imdecode(imgNP,-1)

    videoCap = cv2.VideoCapture(0)

    #videoCap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
    #videoCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
    #videoCap.set(cv2.CAP_PROP_FPS, 25)

    ret, frame = videoCap.read()
    if not ret:
        continue

    # print(frame.shape)
    # frame = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size =  (1024, 768), mean = (0, 0, 0), swapRB = False, crop=False)
    # print(frame.shape)
    # frame = np.transpose(frame.squeeze(), (1, 2, 0))
    # print(frame.shape)
    boxes, confs, points = detector.predict(frame)
    if len(boxes)>0:
        for box in boxes:
            if len(box) > 0:
                x_min, y_min, x_max, y_max = box[0][0], box[0][1], box[0][2], box[0][3]
                x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(frame.shape[1], x_max), min(
                    frame.shape[0], y_max)
                face_save = frame[y_min:y_max, x_min: x_max]
                name_save = '_'.join(re.split('[:.]', f'{datetime.datetime.now()}')) + '.jpg'
                root = 'C:/Users/Buu/PycharmProjects/Camerawebserver/New_Faces/'
                path = os.path.join(root, name_save)
                cv2.imwrite(path, face_save)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))

    cv2.imshow("window", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()