from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

from cleverhans.dataset import MNIST

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# Get MNIST test data
mnist = MNIST(train_start=0, train_end=10000,
              test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')

# Get input for adversarial examples
#x_new = x_train[:1000, :, :, :]
y = y_train[:100]
print(y.shape)

#custom = 'eps=0.5'

folder = 'test' #% custom
#print(folder)
path = "/home/dinhtv/code/adversarial-images/fgsm/%s/%d"
for i in range(0, 10):
	os.makedirs(path % (folder, i))

label = [None] * 100
print(len(label))
for i in range(0, 100):
	for j in range(0, 10):
		if y[i, j] != 0:
			label[i] = j

	os.rename("/home/dinhtv/code/adversarial-images/fgsm/%s/adv_%d.tif" % (folder, i), 
			  "/home/dinhtv/code/adversarial-images/fgsm/%s/%d/adv_%d.tif" % (folder, label[i], i))