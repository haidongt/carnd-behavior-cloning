import csv
import cv2
import numpy as np
import os.path
import random

#directories = ['./data7', './data8']

directories = ['./data7', './data10', './data9']
directories = ['./terrain1', './terrain2', './extra', './extra1']
bucket_number = 10


def loadDirectory(directory, augmented_images, augmented_measurements):
    images = []
    measurements = []
    lines = []
    with open(directory + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (line[0] == 'center'):
                continue
            lines.append(line)

    repeat = {}
    max_bucket_count = 0
    for line in lines:
        measurement = float(line[3])
        bucket = int(measurement * bucket_number)
        if not bucket in repeat:
            repeat[bucket] = 0
        repeat[bucket] = repeat[bucket] + 1

    for key, value in repeat.items():
        if max_bucket_count < value:
            max_bucket_count = value

    for key in repeat:
        repeat[key] = int(max_bucket_count / repeat[key])

    for line in lines:
        source_path = line[0]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = directory + '/IMG/' + filename
        if not os.path.isfile(local_path):
            continue

        measurement = float(line[3])
        bucket = int(measurement * bucket_number)

        keep_prob = 1
        if abs(bucket) == 0:
            keep_prob = 1 #0.015
        if abs(bucket) == 1:
            keep_prob = 1 #0.3
        if abs(bucket) == 2:
            keep_prob = 1 #0.5
        if random.random() > keep_prob:
            continue

        #if measurement == 0:
        #    continue
        '''
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = directory + '/IMG/' + filename
            image = cv2.imread(local_path)
            images.append(image)
   
        correction = 0.2
        measurements.append(measurement)
        measurements.append(measurement + correction)
        measurements.append(measurement - correction)
        '''
        image = cv2.imread(local_path)

        images.append(image)
        measurements.append(measurement)
        #'''
    

    for image, measurement in zip(images, measurements):
         augmented_images.append(image)
         augmented_measurements.append(measurement)
         flipped_image = cv2.flip(image, 1)
         flipped_measurement = -1 *  measurement
         #print(flipped_measurement)
         augmented_images.append(flipped_image)
         augmented_measurements.append(flipped_measurement)


    buckets = {}
    for image, measurement in zip(augmented_images, augmented_measurements):
        print(measurement)
        bucket_key = int(measurement * bucket_number)
        if not bucket_key in buckets:
             buckets[bucket_key] = []
        buckets[bucket_key].append(measurement)
    
    for key, val in buckets.items():
        print(key)
        print(len(val))

images = []
measurements = []

for directory in directories:
    loadDirectory(directory, images, measurements)

# exit()


X_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

from keras.layers.pooling import MaxPooling2D
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

'''
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

'''
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
#model.add(Dropout(0.5))
#model.add(Dense(240))
model.add(Dropout(0.5))
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')



