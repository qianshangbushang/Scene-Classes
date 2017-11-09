from keras.utils import np_utils

import ParseUtil
import SaveUtil
import Net as vgg16
import numpy as np
import os

def load_train_data(path="./train.npz"):
    if os.path.exists(path):
        X, y = np.load(path)['X'], np.load(path)['y']
    else:
        dict = ParseUtil.parse_config()
        print(dict)
        image_base_path = dict['train_image_path']
        label_path = dict['train_label_path']
        label_json_path = dict['train_json_path']
        X, y = SaveUtil.product_npz_data(image_base_path, label_path, label_json_path, path)

    return X, y


def load_test_data(path="./test.npz"):
    if os.path.exists(path):
        X, y = np.load(path)["arr_0"], np.load(path)["arr_1"]
    else:
        dict = ParseUtil.parse_config()
        print(dict)
        image_base_path = dict['test_image_path']
        label_path = dict['test_label_path']
        label_json_path = dict['test_json_path']
        X, y = SaveUtil.product_npz_data(image_base_path, label_path, label_json_path, path)
    return X, y



def main():
    # read data from npz
    #X_train, y_train = load_train_data()
    #X_test, y_test = load_test_data()

    X, y = load_train_data()
    print(X, y)
    X_train = X[:50000]
    y_train = y[:50000]

    X_test = X[50000:]
    y_test = y[50000:]

    X_train = X_train * 1.0 / 255
    X_test = X_test * 1.0 / 255
    y_train_onehot = np.array([vgg16.tran_y(y_train[i]) for i in range(len(y_train))])
    y_test_onehot = np.array([vgg16.tran_y(i) for i in y_test])

    y_train_onehot = np_utils.to_categorical(y_train, 80)
    y_test_onehot = np_utils.to_categorical(y_test, 80)


    #model = vgg16.construct_model_vgg16()
    model = vgg16.construct_model_resnet()

    model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot), epochs=20, batch_size=32)
    scores = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(scores)

if __name__ == '__main__':
    main()