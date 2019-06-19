from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 5
custom = 'iter=20'

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator()

train_dir = '/home/dinhtv/code/adversarial-images/mnisttrainingSet'
validation_dir = '/home/dinhtv/code/adversarial-images/igsm/test'

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(img_rows,img_cols), 
                                                    class_mode='categorical', 
                                                    batch_size=batch_size, 
                                                    color_mode='grayscale',
                                                    seed=None)
validation_generator = validation_datagen.flow_from_directory(validation_dir, 
                                                              target_size=(img_rows,img_cols), 
                                                              class_mode='categorical', 
                                                              batch_size=batch_size, 
                                                              color_mode='grayscale', 
                                                              seed=None)

# training model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 data_format='channels_last',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=400,
                    epochs=epochs,
                    verbose=1)
score = model.evaluate_generator(validation_generator, steps=100, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])