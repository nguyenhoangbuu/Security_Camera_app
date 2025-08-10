import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import urllib.request
import datetime
import re
import os
import dlib

detector = dlib.get_frontal_face_detector()
while True:
    imgresponce = urllib.request.urlopen("http://192.168.1.245/capture?")
    impNp = np.array(bytearray(imgresponce.read()),dtype = np.uint8)
    frame = cv2.imdecode(impNp,-1)
    faces = detector(frame,1)
    if len(faces)>0:
        for face in faces:
            x_min, x_max, y_min, y_max = face.left(), face.right(), face.top(), face.bottom()
            x_min, x_max, y_min, y_max = max(0, x_min), min(frame.shape[1], x_max), max(0,y_min), min(frame.shape[0], y_max)
            cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0))
            face_save = frame[y_min:y_max,x_min:x_max]
            name_save = '_'.join(re.split("[:.]",f"{datetime.datetime.now()}")) + ".jpg"
            root = 'C:/Users/Buu/PycharmProjects/Camerawebserver/New_Faces/'
            path = os.path.join(root,name_save)
            cv2.imwrite(path,face_save)
    cv2.imshow("window",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()