import numpy as np
from sklearn.svm import LinearSVC
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D

class convModels:
    
    def __init__(self, xTrain, yTrain, xTest, yTest):
        self.clf = LinearSVC()
        self.__xTrain = xTrain
        self.__xTest = xTest
        self.__yTrain = yTrain
        self.__yTest = yTest
        self.batch_size = 1
        self.num_classes = 60
        self.epochs = 5
        self.score = None

    def autoencoder(summary=False):
        """ Return the Keras model of the network
            Autoencoder reduces the dimensions of data
            from (56, 64, 24) to 487
        """
        input_shape = (56, 64, 24, 1)
        input_img = Input(shape=input_shape)

        # x = ZeroPadding3D(padding=((2, 3),(2, 1),(1, 0)))(input_img)
        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)

        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)

        encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        decoded = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        if summary:
            print(autoencoder.summary())
        return autoencoder

    def cnn(summary=False):
        """ Return the Keras model of the network
        """
        input_shape = (51, 61, 23, 1)

        model = Sequential()
        model.add(Convolution3D(32, kernel_size=(3, 3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Convolution3D(64, (3, 3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        if summary:
            print(model.summary())
        return model


    def preprocess():
        print("TODO")
    
    def train(self):
        model = cnn()
        train_input_shape = (300, 51, 61, 23, 1)
        test_input_shape = (60, 51, 61, 23, 1)

        x_train = self.__xTrain.reshape(train_input_shape)
        x_test = self.__xTest.reshape(test_input_shape)

        y_train, y_test = get_labels(self.__yTrain, self.__yTest)

        model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

        model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

        self.score = model.evaluate(x_test, y_test, verbose=0)
