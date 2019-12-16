from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_data(rand):
	# loading the data
	images = np.load('images.npy')
	labels = np.load('labels.npy')

	x_train, x_test, y_train, y_test = [], [], [], []

	for i in range(images.shape[0]):
		if i >= 60*rand and i < 60*(rand+1):
			x_test.append(images[i])
			y_test.append(labels[i])
		else:
			x_train.append(images[i])
			y_train.append(labels[i])

	x_train, x_test, y_train, y_test = np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	return x_train, x_test, y_train, y_test

def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	model = Sequential()
	# 1st layer group
	model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
							border_mode='same', name='conv1',
							subsample=(1, 1, 1), 
							input_shape=(3, 16, 112, 112)))
	model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
						   border_mode='valid', name='pool1'))
	# 2nd layer group
	model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
							border_mode='same', name='conv2',
							subsample=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
						   border_mode='valid', name='pool2'))
	model.add(Flatten())
	# FC layers group
	model.add(Dense(4096, activation='relu', name='fc6'))
	model.add(Dropout(.5))
	model.add(Dense(4096, activation='relu', name='fc7'))
	model.add(Dropout(.5))
	model.add(Dense(487, activation='softmax', name='fc8'))
	
	if summary:
		print(model.summary())
	return model

model = get_model(summary=True)

x_train, x_test, y_train, y_test = get_data(2)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
