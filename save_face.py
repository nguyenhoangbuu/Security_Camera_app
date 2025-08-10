import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import urllib.request
from mtcnn import MTCNN
import os
import datetime
import re

detector = MTCNN()
while True:
    imgresponce = urllib.request.urlopen("http://192.168.1.245/capture?")
    imgNp = np.array(bytearray(imgresponce.read()),dtype = np.uint8)
    frame = cv2.imdecode(imgNp,-1)
    # cv2.imshow('Window', frame)
    faces = detector.detect_faces(frame)
    if len(faces) >= 1:
        for face in faces:
            x_min, y_min, width, height = face['box']
            x_max = min(x_min + width, frame.shape[1])
            y_max = min(y_min + height, frame.shape[0])
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            face_save = frame[y_min:y_max, x_min:x_max]
            name_save = '_'.join(re.split("[ :.]",f'{datetime.datetime.now()}')) + '.jpg'
            root = 'C:/Users/Buu/PycharmProjects/Camerawebserver/New_Faces/'
            path = os.path.join(root,name_save)
            cv2.imwrite(path, face_save)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
            cv2.putText(frame, f'{face['confidence']}', (x_min + 100, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f'{name}', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    cv2.imshow('Window', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

# face = cv2.cvtColor(img[y:y + height, x:x + width], cv2.COLOR_RGB2BGR)
#                     name_save = name_root + '_' + str(i) + '.jpg'
#                     root = 'C:/Users/Buu/PycharmProjects/Web_vnsf/Faces'
#                     path = os.path.join(root,name_save)
#                     cv2.imwrite(path, face)