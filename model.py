import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import csv
import cv2
import random
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dropout, Input, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D


X_train = []
y_train = []

with open('/opt/Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for ix, row in enumerate(reader):
        if ix == 0:
            continue
            
        measurement_1 = float(row[3])
        measurement_2 = measurement_1+0.2
        measurement_3 = measurement_1-0.2

        center_image = '/opt/'+row[0].strip()
        left_image = 'data'+row[1].strip()
        right_image = 'data'+row[2].strip()

        X_train.append([center_image, left_image, right_image])
        y_train.append([measurement_1, measurement_2, measurement_3])

X_train, y_train = shuffle(X_train, y_train)
print(X_train[0])
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

print('No. of Training Images:', len(X_train))
print('No. of Validation Images:', len(X_validation))
print('No. of Training Angles:',  len(y_train))
print('No. of Validation Angles:', len(y_validation))
# print(X_train, y_train)

def generator(X_train, y_train, batch_size=32):
    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
   
    while True:
        straight_count = 0
        for i in range(batch_size):
            sample = random.randrange(len(X_train))
            camera_position = random.randrange(3)
            angle = y_train[sample][camera_position]
            if abs(angle) < 0.1:
                straight_count += 1
            
            if straight_count > (batch_size * 0.5):
                while abs(angle) < 0.1:
                    sample = random.randrange(len(X_train))
                    angle = y_train[sample][camera_position]
 
            image = cv2.imread(str(X_train[sample][camera_position]))
            assert image is not None, "Image not found!!"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.float32)
            if np.random.rand()>0.5:
                image = cv2.flip(image, flipCode=1)
                angle = -angle

            images[i] = image
            angles[i] = angle

        yield images, angles

model = Sequential()
model.add(Lambda(lambda x:x / 127.5 - 1, input_shape=(160,320,3)))
model.add((Cropping2D(cropping=((50,20),(0,0)))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, epochs=8, batch_size=32, shuffle=True)
training_info = model.fit_generator(generator(X_train, y_train), steps_per_epoch=len(X_train)/32, validation_steps=len(X_validation)/32, epochs=40, validation_data=generator(X_validation, y_validation), verbose=1)

model.save('model_14.h5')
# print(training_info.history.keys())
# print('Loss:',training_info.history['loss'])
# print('Validation Loss:',training_info.history['val_loss'])






