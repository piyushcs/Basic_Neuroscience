from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn import preprocessing
import keras
import numpy as np

import warnings
warnings.filterwarnings("ignore")

batch_size = 1
global num_classes 
num_classes = 60
epochs = 5

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

def get_labels(y_train, y_test):
	le = preprocessing.LabelEncoder()
	le.fit(y_train)

	y_train = le.transform(y_train)
	y_test = le.transform(y_test)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return y_train, y_test

def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	input_shape = (51, 61, 23, 1)

	model = Sequential()
	model.add(Convolution3D(32, kernel_size=(3, 3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Convolution3D(64, (3, 3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	# model.add(Convolution3D(64, kernel_size=(3, 3, 3),
	# 					activation='relu',
	# 					input_shape=input_shape,
	# 					name='conv2'))

	# model.add(MaxPooling3D(pool_size=(2, 2, 2),
	# 					strides=(2, 2, 2),
	# 					border_mode='valid',
	# 					name='pool2'))

	# model.add(Flatten())
	# # model.add(Dense(4096, activation='relu', name='fc3'))
	# model.add(Dense(num_classes))
	# model.add(Activation('softmax'))

	if summary:
		print(model.summary())
	return model

model = get_model(summary=True)

x_train, x_test, y_train, y_test = get_data(2)

train_input_shape = (300, 51, 61, 23, 1)
test_input_shape = (60, 51, 61, 23, 1)

x_train = x_train.reshape(train_input_shape)
x_test = x_test.reshape(test_input_shape)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train, y_test = get_labels(y_train, y_test)


model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])

model.fit(x_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])