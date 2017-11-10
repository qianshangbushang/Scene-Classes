from keras.applications import VGG16
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from Resnet import *

import numpy as np


def tran_y(y):
    y_onehot = np.zeros(80)
    y_onehot[y] = 1
    return y_onehot

def construct_model_resnet():
    model = ResnetBuilder.build_resnet_50((3, 112, 112), 80)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

def construct_model_vgg16():
    model = VGG16(weights=None, classes=80)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

def construct_model():
    # coding=utf-8
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data("./mnist.npz")
    print(X_train[0].shape)
    print(y_train[0])

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    y_train_onehot = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_test_onehot = np.array([tran_y(i) for i in y_test])

    print("#########################################################")
    print(y_train_onehot)
    print(y_test_onehot)
    print("#########################################################")

    model = VGG16
    print("shape: ", X_train.shape)
    model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=20, batch_size=128)
    scores = model.evaluate(X_test, y_test_onehot, verbose=0)

    print(scores)

if __name__ == '__main__':
    main()
