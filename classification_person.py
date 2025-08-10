import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder


paths = []
labels = []
data = []
for dir_name,_,file_names in os.walk("C:/Users/Buu/PycharmProjects/Camerawebserver/Data_person"):
    if len(file_names) == 0:
        continue
    label = dir_name.split('\\')[-1]
    for file_name in file_names:
        path = os.path.join(dir_name,file_name)
        img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(100,300))/255
        data.append(img)
        paths.append(path)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)
labels = np.expand_dims(labels,1)
enc = OneHotEncoder()
labels = enc.fit_transform(labels)
data = np.array(data, dtype='float32')
labels = labels.toarray()
print(labels.shape)


X_train, X_test, y_train, y_test = train_test_split(data,labels,shuffle=True,stratify=labels,test_size=0.05)

int1 = Input(shape = (300,100,3))
out = Conv2D(64, 3, activation = 'relu')(int1)
out = AveragePooling2D(pool_size = (3,3))(out)
out = Conv2D(128, 3, activation = 'relu')(out)
out = AveragePooling2D(pool_size = (3,3))(out)
out = Conv2D(256, 3, activation = 'relu')(out)
out = Flatten()(out)
out = Dense(256, activation = 'relu')(out)
out = Dropout(rate = 0.5)(out)
out = Dense(128, activation = 'relu')(out)
out = Dropout(0.5)(out)
out = Dense(4, activation = 'softmax')(out)
model = Model(inputs = int1, outputs = out)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy','mse'])

mchp = ModelCheckpoint('model_person.keras',save_best_only = True)
redu = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, min_lr = 0.0000001, patience = 3)
ear = EarlyStopping(monitor = 'val_loss', patience = 11)
model.fit(X_train,y_train, epochs = 23, validation_data = [X_test,y_test], callbacks = [mchp,redu,ear])

# cv2.imshow("Image", X_test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

preds = model.predict(X_test)
labels_predict = enc.inverse_transform(preds)
labels_right = enc.inverse_transform(y_test)
for i in range(len(X_test)):
    img = cv2.putText(X_test[i],labels_predict[i][0] + '-' + labels_right[i][0], (5,25), 2,0.5,(255,0,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

