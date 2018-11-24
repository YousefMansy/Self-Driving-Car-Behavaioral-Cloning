import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

images = []
measurements = []

with open('Data/driving_log_fixed.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print('reading from', row[0])
        center_image = cv2.imread(row[0])
        images.append(center_image)
        measurements.append(float(row[3]))

X_train, y_train = np.array(images), np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')





