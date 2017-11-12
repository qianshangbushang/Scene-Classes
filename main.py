from keras.utils import np_utils

import ParseUtil
import SaveUtil
import Net as vgg16
import numpy as np
import os
import glob
import cv2
import ImagePredeal as IP
from keras.preprocessing.image import img_to_array
from keras.models import  load_model
import json
import heapq

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
        image_result, image_name_arr = np.load(path)["X"], np.load(path)["name"]
    else:
        dict = ParseUtil.parse_config()
        print(dict)
        image_base_path = dict['test_image_path']
        files = glob.glob(image_base_path + "/" + "*.jpg")
        image_result = []
        image_name_arr = []
        for file in files:
            print("deal with image: "+ file)
            image_name = file.strip().split("/")[-1]
            image = cv2.imread(file)
            image = IP.resize(image, 112)
            #cv2.imwrite("../output_file/" + image_name, image)
            image_result.append(img_to_array(image))
            image_name_arr.append(image_name)
        image_result = np.array(image_result)
        image_name_arr = np.array(image_name_arr)
        np.savez(path, X=image_result, name=image_name_arr)
    return image_result, image_name_arr


def predict(path="./test.npz"):
    X, name = load_test_data(path)
    model = load_model("./scene_class_resnet50_model")
    X = X * 1.0 / 255
    result = model.predict(X)
    output = []
    for index, elem in enumerate(result):
        print(name[index], elem.argsort()[-3:][::-1])
        dict ={}
        dict['image_id'] = name[index]
        dict['label_id'] = (elem.argsort()[-3:][::-1]).tolist()
        output.append(dict)
    with open("output.json", "w") as f:
        json.dump(output, f)
    print("输出完成！")
    #print(result)




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
    model.save("scene_class_resnet50_model")


if __name__ == '__main__':
    #main()
    predict()