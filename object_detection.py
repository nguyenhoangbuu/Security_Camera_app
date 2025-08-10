import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from ultralytics import YOLO
import cv2
import os
import datetime
import re
import telegram
import asyncio
import nest_asyncio

nest_asyncio.apply()

my_token = "XXX"

# Tạo bot
bot = telegram.Bot(token=my_token)

detector = YOLO('yolov8n.pt')
root = "C:/Users/Buu/PycharmProjects/Camerawebserver/New_Faces/"
while True:
    responce = urllib.request.urlopen("http://192.168.1.245/capture?")
    imgNp = np.array(bytearray(responce.read()),np.uint8)
    frame = cv2.imdecode(imgNp,-1)
    # frame = cv2.dnn.blobFromImage(frame,
    #                               scalefactor=1 / 255,
    #                               size=(1024, 768),
    #                               swapRB=True)
    # frame = np.transpose(frame.squeeze(), (1, 2, 0))
    results = detector.track(frame)
    print(type(results))
    print(len(results))
    print(type(results[0]))
    print(len(results[0]))
    # print(results)
    if len(results)>0:
        for result in results:
            classes_name = result.names
            for box in result.boxes:
                cls = int(box.cls[0])
                name = classes_name[cls]
                # print(name)
                if name == 'person' and box.conf[0] > 0.7:
                    print(box.conf[0])
                    #asyncio.run(bot.sendMessage(chat_id="6940850899", text="Gưi từ PyCharm"))
                    [x_min, y_min, x_max, y_max] = box.xyxy[0]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    x_min, x_max, y_min, y_max = max(0, x_min), min(frame.shape[1], x_max), max(0, y_min), min(frame.shape[0], y_max)
                    # bot.sendPhoto(chat_id="6940850899", photo = frame[y_min:y_max,x_min:x_max], caption = "Phát hiện chuyển động!")
                    #bot.sendPhoto(chat_id="5115269968", photo=frame[y_min:y_max, x_min:x_max], caption="Phát hiện chuyển động!")
                    name_object = "_".join(re.split("[:.]",f'{datetime.datetime.now()}')) + '.jpg'
                    path = os.path.join(root,name_object)
                    cv2.imwrite(path,frame[y_min:y_max,x_min:x_max])
                    asyncio.run(bot.sendPhoto(chat_id="XXX", photo=open(path, "rb"), caption="Phát hiện chuyển động!"))
                    asyncio.run(bot.sendPhoto(chat_id="", photo=open(path, "rb"), caption="Phát hiện chuyển động!"))
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0))
                    cv2.putText(frame, name,(x_min+5,y_min+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    cv2.imshow("window",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()